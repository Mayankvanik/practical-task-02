from google import genai
from google.genai import types


def test_gemini():
    try:
        # Initialize client (auto picks GOOGLE_API_KEY)
        client = genai.Client(api_key="AIzaSyBsXRJgwgH_uh_OFS6Jk0TiNZQwC5AMLrA")

        # Simple test query
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview",
            contents="Explain what is RAG in 2 lines."
        )

        print("✅ API Key is working!")
        print("\nResponse:\n")
        print(response.text)

    except Exception as e:
        print("❌ API Key or request failed")
        print("Error:", str(e))


if __name__ == "__main__":
    test_gemini()


# from langchain_google_genai import ChatGoogleGenerativeAI

# def test_gemini_langchain():
#     try:
#         llm = ChatGoogleGenerativeAI(
#             api_key="AIzaSyBsXRJgwgH_uh_OFS6Jk0TiNZQwC5AMLrA",
#             model="gemini-2.5-flash",
#             temperature=0.3
#         )

#         response = llm.invoke("Explain RAG in 2 lines.")

#         print("✅ API Key is working!\n")
#         print(response.content)

#     except Exception as e:
#         print("❌ Error:", str(e))


# if __name__ == "__main__":
#     test_gemini_langchain()