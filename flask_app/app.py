from flask import Flask, request, jsonify
import logging
from utils import search_articles, concatenate_content, generate_answer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        user_query = data.get('query')

        if not user_query:
            logger.warning("No query provided in the request.")
            return jsonify({"error": "No query provided."}), 400

        logger.info("Received query: %s", user_query)

        # Step 1: Search articles
        articles = search_articles(user_query)
        if not articles:
            logger.warning("No articles found for query: %s", user_query)
            return jsonify({"error": "No articles found."}), 404

        # Step 2: Concatenate article contents
        content = concatenate_content(articles)
        if not content:
            logger.error("Failed to fetch article content for query: %s", user_query)
            return jsonify({"error": "Failed to fetch article content."}), 500

        # Step 3: Generate answer from content
        answer = generate_answer(content, user_query)
        if not answer:
            logger.error("Failed to generate answer for query: %s", user_query)
            return jsonify({"error": "Failed to generate answer."}), 500

        logger.info("Successfully generated answer for query: %s", user_query)
        return jsonify({"answer": answer})

    except Exception as e:
        logger.exception("Unexpected error occurred while processing query.")
        return jsonify({"error": "Internal server error."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
