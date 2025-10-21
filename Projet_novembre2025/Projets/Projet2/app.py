from flask import Flask, render_template

# Initialize the Flask application
app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    # Data to pass to the frontend
    title = "Simple Python Web App"
    items = ["Item 1", "Item 2", "Item 3"]
    
    # Render the HTML template, passing the data to it
    return render_template('index.html', page_title=title, list_items=items)

# Run the application
if __name__ == '__main__':
    # You would typically use a production server like Gunicorn, 
    # but for development, this works.
    app.run(debug=True)