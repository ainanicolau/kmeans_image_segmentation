#!/usr/bin/env python3

from http.server import HTTPServer, SimpleHTTPRequestHandler
import kmeans as km
import cgi
import base64
from io import BytesIO
from PIL import Image

PORT = 8080


class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        # Print message to console indicating a POST request has been received
        print("Receiving post")

        # Create environment variable with request information
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": self.headers["Content-Type"],
        }

        # Parse the form data in the request
        form = cgi.FieldStorage(
            fp=self.rfile, headers=self.headers, environ=environ
        )

        # Get the uploaded image bytes from the form data
        image_bytes = form.getvalue("image")

        # Open the image from the bytes using the Image module
        image = Image.open(BytesIO(image_bytes))

        # Set the response headers and status code
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        # Process the uploaded image using k-means with k=10
        segmented_im = km.run(image, 10)

        # Convert the segmented image to a base64-encoded JPEG image
        buffer_ = BytesIO()
        segmented_im.save(buffer_, format="JPEG")
        myimage = buffer_.getvalue()
        im64 = base64.b64encode(myimage)

        # Load the HTML response template from a file
        with open("response.html", "r") as file:
            data = file.read()

        # Embed the base64-encoded image in the HTML response
        im_src = '"data:image/jpeg;base64, {}"'.format(str(im64)[2:-1])
        data = (data.format(im_src)).encode()

        # Write the HTML response to the client
        self.wfile.write(data)


def main():
    """
    Runs a server and listens to requests for image segmentation.
    """
    # Create an HTTP server with the specified port and request handler
    httpd = HTTPServer(("", PORT), MyHTTPRequestHandler)

    # Print a message indicating that the server is running
    print("Serving at port " + str(PORT))

    # Start the server and keep it running indefinitely
    httpd.serve_forever()


if __name__ == "__main__":
    main()
