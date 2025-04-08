from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="f7kWJaMjaV59ujyAkdoD"
)

result = CLIENT.infer(r'C:/Users/mohanraj/Desktop/SAR images dataset from google/image_249.png', model_id="-biv3n/1")
