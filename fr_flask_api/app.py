from flask import Flask, request, jsonify
import os
import boto3
from dotenv import load_dotenv
from urllib.parse import urlparse

dotenv_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), '..', '.env')

load_dotenv(dotenv_path)

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
data_folder = os.path.join(parent_directory, 'data') 

app = Flask(__name__)

UPLOAD_FOLDER = data_folder + '/users_data/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def create_user_folder(name, email, user_id):
    folder_name = f"{name}--{email}--{user_id}"
    folder_path = app.config['UPLOAD_FOLDER'] + folder_name
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def delete_user_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(folder_path)


aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')


if not aws_access_key_id or not aws_secret_access_key:
    raise ValueError("AWS credentials are not set in environment variables.")

s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key)


@app.route('/api/v1/create-contract-worker', methods=['POST'])
def create_contract_worker():
    user_id = request.form.get('user_id')
    name = request.form.get('name')
    email = request.form.get('email')
    image_urls_s3_string = request.form.get('image_urls_s3', '')
    image_urls_s3 = image_urls_s3_string.split(',') if image_urls_s3_string else []
    user_folder = create_user_folder(name, email, user_id)
    for image_url in image_urls_s3:
        parsed_url = urlparse(image_url)
        image_key = parsed_url.path.lstrip('/')
        image_filename = os.path.basename(image_key)
        image_path = user_folder + '/' + image_filename

        s3.download_file(aws_bucket_name,
                         image_key, image_path)


    return jsonify({'success': 'Images downloaded and saved successfully'}), 200


@app.route('/api/v1/update-contract-worker', methods=['POST'])
def update_contract_worker_images():
    user_id = request.form.get('user_id')
    name = request.form.get('name')
    email = request.form.get('email')
    image_urls_s3_string = request.form.get('image_urls_s3', '')
    image_urls_s3 = image_urls_s3_string.split(
        ',') if image_urls_s3_string else []

    user_folder = create_user_folder(name, email, user_id)

    for filename in os.listdir(user_folder):
        file_path = user_folder + '/' + filename
        if os.path.isfile(file_path):
            os.remove(file_path)

    image_urls_s3_string = request.form.get('image_urls_s3', '')
    image_urls_s3 = image_urls_s3_string.split(
        ',') if image_urls_s3_string else []

    for image_url in image_urls_s3:
        parsed_url = urlparse(image_url)
        image_key = parsed_url.path.lstrip('/')
        image_filename = os.path.basename(image_key)
        image_path = os.path.join(user_folder, image_filename)
        image_path = user_folder+ '/' +  image_filename

        s3.download_file(aws_bucket_name, image_key, image_path)

    return jsonify({'success': 'Images updated successfully'}), 200


@app.route('/api/v1/delete-contract-worker/<int:user_id>', methods=['DELETE'])
def delete_contract_worker(user_id):
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        user_folder = app.config['UPLOAD_FOLDER'] + '/' +  f"{name}-{email}-{user_id}"
        delete_user_folder(user_folder)
        return jsonify({'success': 'Contract worker folder deleted successfully'}), 200
    except Exception as e:
        error_message = str(e)
        return jsonify({'error': f'Failed to delete contract worker folder: {error_message}'}), 500


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")