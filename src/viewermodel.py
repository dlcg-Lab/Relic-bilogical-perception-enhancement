import json
import cProfile
import openai
import string
from deepdanbooru_onnx import DeepDanbooru


class ViewerModel:
    def __init__(self,
                 threshold=0.5,
                 batch_size=1,
                 debug=False,
                 decay_rate=0.5,
                 removal_threshold=0.1,
                 llm="gpt-3.5-turbo"
                 ):

        # Set OpenAI API configuration
        openai.api_base = 'https://api.closeai-proxy.xyz/v1'
        openai.api_key = 'sk-WYkFRLZwB9vBzh5RLusi7VuA1HtSt496Wvc5PyVO1SD2ApSa'

        # Initialize DeepDanbooru with given parameters
        self.deep_danbooru = DeepDanbooru(mode='auto', threshold=threshold, batch_size=batch_size)

        # Store previous frame's analyzed label and confidence
        self.pre_str = ""
        self.str_debug = debug

        # Initialize keywords dictionary
        self.keywords = {}

        # Add decay rate and removal threshold as class members
        self.decay_rate = decay_rate
        self.removal_threshold = removal_threshold

        self.llm = llm

    def update_keywords(self):
        # Update the weights of the keywords by subtracting the decay_rate
        for key in list(self.keywords.keys()):
            self.keywords[key] *= self.decay_rate
            if self.keywords[key] < self.removal_threshold:
                del self.keywords[key]
        if self.str_debug:
            print("current keywords:", self.keywords)

    def deep_danbooru(self, image_path):
        # Helper function to get DeepDanbooru predictions on a single image
        js = self.deep_danbooru(image_path)
        for item, weight in js.items():
            if item in self.keywords:
                self.keywords[item] += weight
                self.keywords[item] = min(1.0, self.keywords[item])
            else:
                self.keywords[item] = weight
        return js

    def message_constructor(self, current_keywords):
        # Convert keywords dictionary into a readable string for both self.keywords and current_keywords
        self_keywords_str = ', '.join([f"{key}: {value}" for key, value in self.keywords.items()])
        current_keywords_str = ', '.join([f"{key}: {value}" for key, value in current_keywords.items()])

        # Construct the prompt for GPT API
        message = f"The previous frame was described as: '{self.pre_str}'. " \
                  f"Based on the previous keywords: {self_keywords_str} and current keywords: {current_keywords_str}, " \
                  "if you infer that the current scene is similar to the previous frame, " \
                  "return the previous description without any changes. " \
                  "Otherwise, provide a new description for the current frame."

        if self.str_debug:
            print("message:", message)
        return message

    def gpt_output_str(self, current_keywords):
        # Helper function that sends a prompt to the GPT API and returns its response
        message = self.message_constructor(current_keywords)
        response = openai.ChatCompletion.create(
            model=self.llm,
            messages=[
                {"role": "system",
                 "content": "Based on the provided information, describe the scene using a phrase or sentence of no more than 5 words. Do not include any reference to humans and animals, human-like figures, or any form or implication of a person's presence. Also, avoid using any adjectives."
                 },
                {"role": "user",
                 "content": message
                 },
            ]
        )
        output = response["choices"][0]["message"]["content"]
        self.pre_str = output
        return output

    def remove_punctuation(self, s):
        # 使用translate方法移除字符串中的所有标点符号
        return s.translate(str.maketrans('', '', string.punctuation))

    def pipeline_str(self, image_path):
        current_keywords = self.deep_danbooru(image_path)
        output_desc = self.gpt_output_str(current_keywords)
        output_desc = self.remove_punctuation(output_desc)  # 调用remove_punctuation方法
        self.update_keywords()
        return output_desc

    def debug(self, image_path):
        # Debug function to profile the pipeline execution time
        cProfile.runctx('self.pipeline(image_path)', globals(), locals(), sort='cumulative')


if __name__ == '__main__':
    # Instantiate the model and call the pipeline function with an example image
    model = ViewerModel(threshold=0.2)
    print(model.pipeline_str("../resources/input.jpg"))
