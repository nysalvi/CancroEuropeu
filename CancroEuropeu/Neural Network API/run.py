from api import app
from waitress import serve

if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=7777)
    serve(app, host='localhost', port=7777)