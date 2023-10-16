import json
import cProfile
from deepdanbooru_onnx import DeepDanbooru
import os
import openai


class Model:
    def __init__(self, threshold=0.5, batch_size=1):
        openai.api_base = 'https://api.closeai-proxy.xyz/v1'
        openai.api_key = 'sk-WYkFRLZwB9vBzh5RLusi7VuA1HtSt496Wvc5PyVO1SD2ApSa'

        # Initialize DeepDanbooru class with given parameters
        self.deep_danbooru = DeepDanbooru(threshold=threshold, batch_size=batch_size)

        # Keep track of the previous frame's analyzed label and confidence
        self.pre_str = ""

    def deep_danbooru_str(self, image_path):
        # Helper function to get DeepDanbooru predictions on a single image
        js = self.deep_danbooru(image_path)
        return str(js)

    def message_constructor(self, str1):
        # Helper function to construct the prompt to send to GPT-3 API
        message = "There is a video with the following label and confidence for the previous frame: "
        message += self.pre_str
        message += ". And for the current frame, the following label and confidence are: "
        message += str1
        message += ". These two frames are consecutive. Please guess the scene in which this " \
                   "image is located based on this information."
        return message

    def pipeline(self, image_path):
        # Main pipeline function that takes an image path as input and outputs a JSON object
        str1 = self.deep_danbooru_str(image_path)

        # Save current label and confidence to use for next frame
        self.pre_str = str1

        # Use GPT-3 API to generate the scene description based on previous and current frame visual data
        str2 = self.gpt(str1)

        # Transform the GPT-3 output text into a JSON object
        return self.json_transformer(str2)

    def gpt(self, str1):
        # Helper function that sends a prompt to the GPT-3 API and returns its response
        message = self.message_constructor(str1)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Output a json file in the following format:"
                            "{"
                            "indoor or outdoor: str, "
                            "weather: str, scene: str, "
                            "objects: [str], color: str, "
                            "feeling: str"
                            "people counts: int"
                            "}"
                 },
                {"role": "user",
                 "content": message
                 },
            ]
        )
        output = response["choices"][0]["message"]["content"]

        return output

    def json_transformer(self, gpt_output):
        # Helper function that transforms GPT-3 output text into a JSON object
        try:
            json_dict = json.loads(gpt_output)
            if isinstance(json_dict, dict):
                return json_dict
            else:
                return {}
        except json.JSONDecodeError:
            print("Error!")
            return {}

    def debug(self, image_path):
        # Debug function that profile the pipeline execution time
        cProfile.runctx('self.pipeline(image_path)', globals(), locals(), sort='cumulative')


if __name__ == '__main__':
    # Instantiate the model and call the pipeline function with an example image
    model = Model(threshold=0.2)
    print(model.pipeline("../resources/input.jpg"))
