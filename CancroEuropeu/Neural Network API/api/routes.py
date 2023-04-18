from flask import request, jsonify
from flask_restx import Api, Resource

from services.prediction import get_prediction
from werkzeug.datastructures import FileStorage

rest_api = Api(version="1.0", title="Neural Network API", default="Neural Network", default_label="Convolutional Analysis")

parser = rest_api.parser()
parser.add_argument('file', type=FileStorage, location='files')

"""
    Flask-Restx routes
"""

@rest_api.route('/api/request-analysis', doc={"description": 'Analysis Images In Convolutional Neural Network'}, methods=['POST'])
class Items(Resource):

    @rest_api.expect(parser)
    def post(self):
        """
            Returns if a plant is infected with Canker
        """
        
        if request.method == 'POST':
            file = request.files['file']
            img_bytes = file.read()
            class_id, class_name = get_prediction(image_bytes=img_bytes)
            return jsonify({'class_id': class_id, 'class_name': class_name})
        
        return {"success": False}, 405
