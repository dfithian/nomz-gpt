import base64
from collections import namedtuple
import os

from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI()

MODEL = 'gpt-4o-mini'

def analyze_text(text):
    prompt = f"Analyze the following text and extract ingredients and steps from the recipe. Return a JSON object with two fields: `ingredients` should be a list of strings, and `steps` should be a list of strings."
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an assistant that extracts ingredients and steps from recipe images."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": text}
            ]}
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

def analyze_image(image_type, image_base64):
    prompt = f"Analyze the following base64-encoded image and extract ingredients and steps from the recipe. Return a JSON object with two fields: `ingredients` should be a list of strings, and `steps` should be a list of strings."
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an assistant that extracts ingredients and steps from recipe images."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{image_type};base64,{image_base64}",
                    "detail": "low",
                } }
            ]}
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Missing file.'}), 400
    if file.mimetype.startswith('image/'):
        try:
            return analyze_image(file.mimetype, base64.b64encode(file.stream.read()))
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    elif file.mimetype.startswith('text/'):
        try:
            if file.mimetype == 'text/html':
                # work around token limit by extracting the body
                soup = BeautifulSoup(file.stream.read(), 'html.parser')
                content = str(soup.find('body'))
            else:
                content = str(file.stream.read())
            return analyze_text(content)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Unsupported MIME type.'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
