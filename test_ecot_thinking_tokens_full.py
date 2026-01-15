from google import genai
client = genai.Client(api_key="AIzaSyAVWZb6O3wNWyRz93CjH_RHDfB0o5e9Jyc")
genai.configure(api_key='AIzaSyAVWZb6O3wNWyRz93CjH_RHDfB0o5e9Jyc')
response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)