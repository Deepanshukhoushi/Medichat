# import google.generativeai as genai

# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Load API key
# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     raise ValueError("❌ GEMINI_API_KEY not found in environment!")

# genai.configure(api_key=api_key)

# # Try generating a simple response
# model = genai.GenerativeModel("gemini-1.5-flash-latest")
# response = model.generate_content("Hello Gemini, are you working?")

# print("✅ API Response:")
# print(response.text)




import cohere

# Replace this with your actual Cohere API key
api_key = "EpHBHNDwrbeNiab06FRrWC5VLlTOwHJXaGM2gXVe"

try:
    co = cohere.Client(api_key)

    # Use the Chat API instead of generate()
    response = co.chat(
        model="command-a-03-2025",  # ✅ valid modern model
        message="Hello! This is a test message from my Cohere API key."
    )

    print("✅ Cohere API Key is valid and working!")
    print("Response:", response.text)

except Exception as e:
    print("❌ Error occurred while testing the Cohere API key.")
    print("Details:", e)
