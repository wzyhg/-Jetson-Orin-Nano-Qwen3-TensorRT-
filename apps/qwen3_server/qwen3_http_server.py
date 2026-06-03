import argparse
import json
import queue
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ALLOWED_TARGETS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "dog",
    "cat",
    "bottle",
    "cup",
    "cell phone",
    "laptop",
    "chair",
    "book",
    "clock",
    "any",
]


def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def strip_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def final_answer_text(text):
    is_worker_output = "__KUIPER_META__" in text
    text = re.sub(r"^__KUIPER_META__.*$", "", text, flags=re.MULTILINE)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    elif not is_worker_output:
        lines = text.splitlines()
        text = "\n".join(lines[1:]) if len(lines) > 1 else text

    text = re.sub(r"\nsteps:.*", "", text, flags=re.DOTALL)
    text = strip_think_blocks(text)
    return text.strip()


def extract_allowed_target(text):
    cleaned = normalize_text(text)
    matches = []
    for target in ALLOWED_TARGETS:
        pattern = r"(?<![a-z])" + re.escape(target) + r"(?![a-z])"
        if re.search(pattern, cleaned):
            matches.append(target)
    if not matches:
        return None
    return max(matches, key=len)


def parse_qwen_output(text):
    answer = final_answer_text(text)
    json_match = re.search(r"\{.*\}", answer, flags=re.DOTALL)
    if json_match:
        try:
            plan = json.loads(json_match.group(0))
            action = plan.get("action")
            if action == "track":
                target = plan.get("target_class")
                if target in {"person", "bottle"}:
                    return {"action": "track", "target_class": target}
            if action == "enable":
                return {"action": "enable", "enabled": bool(plan.get("enabled", False))}
            if action == "pause_when_found":
                target = plan.get("target_class")
                if target in ALLOWED_TARGETS:
                    return {"action": "track", "target_class": target}
            if action == "unknown":
                return {"action": "unknown"}
        except json.JSONDecodeError:
            pass

    normalized = normalize_text(answer)
    target = extract_allowed_target(answer)
    if target in {"person", "bottle"}:
        return {"action": "track", "target_class": target}
    if re.search(r"(?<![a-z])stop(?![a-z])", normalized):
        return {"action": "enable", "enabled": False}
    if re.search(r"(?<![a-z])resume(?![a-z])", normalized):
        return {"action": "enable", "enabled": True}
    return {"action": "unknown"}


def extract_generation_stats(text):
    meta_match = re.search(
        r"__KUIPER_META__\s+steps:(\d+)\s+duration:([0-9.]+)\s+steps/s:([0-9.]+)",
        text,
    )
    if meta_match:
        return {
            "steps": int(meta_match.group(1)),
            "duration": float(meta_match.group(2)),
            "steps_per_second": float(meta_match.group(3)),
        }

    steps_match = re.search(r"\nsteps:(\d+)", text)
    duration_match = re.search(r"\nduration:([0-9.]+)", text)
    rate_match = re.search(r"\nsteps/s:([0-9.]+)", text)
    if steps_match and duration_match and rate_match:
        return {
            "steps": int(steps_match.group(1)),
            "duration": float(duration_match.group(1)),
            "steps_per_second": float(rate_match.group(1)),
        }
    return {}


def _legacy_build_prompt(command):
    return (
        "/no_think\n"
        "你是机器人云台命令分类器。只输出一个英文标签，不要解释。\n"
        "标签只能是: stop, resume, person, bottle, cup, car, bus, truck, dog, cat, traffic light, stop sign, any\n"
        "先判断控制命令: 停止、暂停、停一下、暂停一下、别跟踪、不要跟踪 => stop。\n"
        "继续、恢复、开始、继续跟踪、恢复跟踪 => resume。\n"
        "如果不是控制命令，再判断目标: 找人、跟踪人、看人 => person；找瓶子、瓶子 => bottle；红绿灯、交通灯 => traffic light。\n"
        "示例: 暂停一下=>stop；停止跟踪=>stop；继续跟踪=>resume；找人=>person；找瓶子=>bottle。\n"
        "当前命令: "
        + command
        + "\n输出标签:"
    )

def build_prompt(command):
    return (
        "/no_think\n"
        "Output JSON only. Choose exactly one:\n"
        "找人/追踪人/跟踪人 => {\"action\":\"track\",\"target_class\":\"person\"}\n"
        "找瓶子/追踪瓶子/跟踪瓶子 => {\"action\":\"track\",\"target_class\":\"bottle\"}\n"
        "停止/暂停/停一下 => {\"action\":\"enable\",\"enabled\":false}\n"
        "继续/恢复/开始 => {\"action\":\"enable\",\"enabled\":true}\n"
        "Command: "
        + command
        + "\nJSON:"
    )


class Qwen3Parser:
    def __init__(self, executable, model, tokenizer, timeout, int8=False, persistent=True,
                 max_steps=128):
        self.executable = executable
        self.model = model
        self.tokenizer = tokenizer
        self.timeout = timeout
        self.int8 = int8
        self.persistent = persistent
        self.max_steps = max_steps
        self.lock = threading.Lock()
        self.proc = None
        self.lines = None
        self.reader = None
        if self.persistent:
            self.start_worker()

    def build_cmd(self, prompt=None, server=False):
        cmd = [self.executable, self.model, self.tokenizer]
        if prompt is not None:
            cmd.append(prompt)
        if self.int8:
            cmd.append("--int8")
        cmd.extend(["--max-steps", str(self.max_steps)])
        if server:
            cmd.append("--server")
        return cmd

    def start_worker(self):
        cmd = self.build_cmd(server=True)
        self.lines = queue.Queue()
        self.proc = subprocess.Popen(
            cmd,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        def read_stdout():
            for line in self.proc.stdout:
                self.lines.put(line.rstrip("\n"))
            self.lines.put(None)

        self.reader = threading.Thread(target=read_stdout, daemon=True)
        self.reader.start()

        while True:
            try:
                line = self.lines.get(timeout=self.timeout)
            except queue.Empty as exc:
                raise TimeoutError("qwen3 worker startup timeout") from exc
            if line is None:
                raise RuntimeError("qwen3 worker exited before ready")
            if line == "__KUIPER_READY__":
                return

    def request_worker(self, prompt):
        if self.proc.poll() is not None:
            raise RuntimeError(f"qwen3 worker exited with code {self.proc.returncode}")

        safe_prompt = re.sub(r"[\r\n]+", " ", prompt).strip()
        self.proc.stdin.write(safe_prompt + "\n")
        self.proc.stdin.flush()

        output = []
        in_body = False
        while True:
            try:
                line = self.lines.get(timeout=self.timeout)
            except queue.Empty as exc:
                raise TimeoutError("qwen3 worker request timeout") from exc
            if line is None:
                raise RuntimeError("qwen3 worker exited during request")
            if line == "__KUIPER_BEGIN__":
                output = []
                in_body = True
                continue
            if line == "__KUIPER_END__":
                return "\n".join(output)
            if in_body:
                output.append(line)

    def parse(self, command):
        prompt = build_prompt(command)
        with self.lock:
            if self.persistent:
                raw = self.request_worker(prompt)
                returncode = 0
            else:
                result = subprocess.run(
                    self.build_cmd(prompt=prompt),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                )
                raw = result.stdout
                returncode = result.returncode
        parsed = parse_qwen_output(raw)
        parsed.update(extract_generation_stats(raw))
        return parsed


def make_handler(parser):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"ok": True})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/parse":
                self._send_json(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                command = str(payload.get("command", "")).strip()
                if not command:
                    self._send_json(400, {"error": "missing command"})
                    return
                self._send_json(200, parser.parse(command))
            except (subprocess.TimeoutExpired, TimeoutError):
                self._send_json(504, {"error": "qwen3 timeout"})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args))

    return Handler


def main():
    arg_parser = argparse.ArgumentParser(description="Qwen3 HTTP command parser.")
    arg_parser.add_argument("--host", default="127.0.0.1")
    arg_parser.add_argument("--port", type=int, default=18080)
    arg_parser.add_argument("--executable", default="/workspaces/KuiperLLama/build-qwen3/demo/qwen3_infer")
    arg_parser.add_argument("--model", default="/models/qwen3-0.6b/qwen3-int8.bin")
    arg_parser.add_argument("--tokenizer", default="/models/qwen3-0.6b/tokenizer.json")
    precision_group = arg_parser.add_mutually_exclusive_group()
    precision_group.add_argument("--int8", dest="int8", action="store_true", help="Run qwen3_infer with --int8.")
    precision_group.add_argument("--fp32", dest="int8", action="store_false", help="Run qwen3_infer without --int8.")
    arg_parser.set_defaults(int8=True)
    arg_parser.add_argument("--subprocess-per-request", action="store_true",
                            help="Start qwen3_infer for every request instead of keeping a worker alive.")
    arg_parser.add_argument("--max-steps", type=int, default=128)
    arg_parser.add_argument("--timeout", type=float, default=120.0)
    args = arg_parser.parse_args()

    parser = Qwen3Parser(
        args.executable,
        args.model,
        args.tokenizer,
        args.timeout,
        args.int8,
        persistent=not args.subprocess_per_request,
        max_steps=args.max_steps,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(parser))
    mode = "persistent" if not args.subprocess_per_request else "subprocess-per-request"
    print(f"Qwen3 parser server listening on http://{args.host}:{args.port} ({mode})")
    server.serve_forever()


if __name__ == "__main__":
    main()
