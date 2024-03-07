import faiss
import time
import numpy as np
import pickle
def create_faiss_index(embeddings):
    start_time = time.time()
    D = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(D)
    embeddings_array = np.vstack(embeddings).astype('float32')
    index.add(embeddings_array)
    end_time = time.time()
    print(f"Total time to generate FAISS database: {end_time - start_time} seconds")
    return index
def read_data():
    f = open("main_category_embeddings.pkl", "rb")
    main_category = pickle.load(f)
    f.close()
    
    # Finding mean of main category:
    embeddings = []
    index = 0
    for key, value in main_category.items():
        sum = 0
        # Find all the embeddings
        emb = []
        text_file_index = 0
        for v in value:
            emb.append(v[0])
            # Save the 1 column as text file
            np.save(f'ecom_npy_files/{index}-{text_file_index}.npy',v[1])
            text_file_index += 1
            sum += v[0]
        temp = create_faiss_index(emb)
        faiss.write_index(temp, f'ecom_embeddings/{index}.index')
        index += 1
        embeddings.append(sum/len(value))
        
    # print(embeddings[0])
    index = create_faiss_index(embeddings)
    faiss.write_index(index, 'main.index')
    # print(index)

if __name__ == "__main__":
    read_data()