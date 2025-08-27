import google.generativeai as genai
import os

# Load API key
api_key = "AIzaSyC1e813j2aM00E9nZPN-QJjWt6EgNxWldM"
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in environment!")

genai.configure(api_key=api_key)

# Try generating a simple response
model = genai.GenerativeModel("gemini-1.5-flash")  # use flash for quick test
response = model.generate_content("Hello Gemini, are you working?")

print("✅ API Response:")
print(response.text)