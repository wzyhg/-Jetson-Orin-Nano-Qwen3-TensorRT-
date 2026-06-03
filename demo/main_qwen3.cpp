#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/qwen3.h"

struct GenerateResult {
  int32_t steps = 0;
  std::string text;
};

GenerateResult generate(const model::Qwen3Model& model, const std::string& sentence,
                        int max_new_tokens) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";
  const int32_t total_steps = prompt_len + max_new_tokens;

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
      if (next != 151645 && next != 151644) {
        words.push_back(next);
      }
    }
    if (!is_prompt && model.is_sentence_ending(next)) {
      break;
    }

    if (is_prompt) {
      next = tokens.at(pos + 1);
    }
    pos += 1;
  }
  return GenerateResult{std::min(pos, total_steps), model.decode(words)};
}

std::string fill_template(const std::string& content) {
  const std::string format =
      "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
  std::string result = format;
  size_t pos = result.find("%s");
  if (pos != std::string::npos) {
    result.replace(pos, 2, content);
  }
  return result;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    LOG(INFO) << "Usage: ./qwen3_infer checkpoint_path tokenizer_path [prompt] [--int8] "
                 "[--server] [--max-steps N]";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];
  bool is_int8_model = false;
  bool server_mode = false;
  int max_new_tokens = 2560;
  std::string hi = "What is AI?";
  bool prompt_set = false;
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--int8") {
      is_int8_model = true;
    } else if (arg == "--server") {
      server_mode = true;
    } else if (arg == "--max-steps") {
      if (i + 1 >= argc) {
        LOG(ERROR) << "--max-steps requires a value";
        return -1;
      }
      max_new_tokens = std::stoi(argv[++i]);
    } else if (!prompt_set) {
      hi = arg;
      prompt_set = true;
    } else {
      LOG(ERROR) << "Unknown option: " << arg;
      return -1;
    }
  }
  std::string checkpoint(checkpoint_path);
  if (checkpoint.find("q4") != std::string::npos || checkpoint.find("Q4") != std::string::npos) {
    LOG(ERROR) << "Qwen3 Q4 checkpoint is not supported by this executable yet. "
               << "Use qwen0.6.bin2, or implement a Q4 loader/kernel before running "
               << checkpoint;
    return -1;
  }

  model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path,
                          is_int8_model);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code()
               << ", message: " << init_status.get_err_msg();
  }

  if (server_mode) {
    std::cout << "__KUIPER_READY__" << std::endl;
    std::string request;
    while (std::getline(std::cin, request)) {
      if (request == "__KUIPER_QUIT__") {
        break;
      }
      const std::string sentence = fill_template(request);
      auto start = std::chrono::steady_clock::now();
      GenerateResult result = generate(model, sentence, max_new_tokens);
      auto end = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration<double>(end - start).count();
      std::cout << "__KUIPER_BEGIN__" << std::endl;
      std::cout << result.text << std::endl;
      std::cout << "__KUIPER_META__ steps:" << result.steps << " duration:" << duration
                << " steps/s:" << static_cast<double>(result.steps) / duration << std::endl;
      std::cout << "__KUIPER_END__" << std::endl;
    }
    return 0;
  }

  std::cout << hi << "\n";
  const std::string& sentence = fill_template(hi);
  auto start = std::chrono::steady_clock::now();
  fflush(stdout);
  GenerateResult result = generate(model, sentence, max_new_tokens);
  printf("%s ", result.text.data());
  fflush(stdout);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps:%d\n", result.steps);
  printf("\nduration:%lf\n", duration);
  printf("\nsteps/s:%lf\n", static_cast<double>(result.steps) / duration);
  fflush(stdout);
  return 0;
}
