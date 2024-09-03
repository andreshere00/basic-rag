from rag import Main

if __name__ == "__main__":
    query = "What is about the text?"
    file_path = "./examples/example.pdf"
    model_name = "text-embedding-ada-002"
    length = 8192
    method = "fixed"

    main_process = Main()
    main_process.execute(query, file_path, model_name, length, method)
