# Move Prediction in Gomoku Using Deep Learning(CNN)

## Introduction
- It is a Convolutional Neural Network model which was trained with Renju Database. It is a fun project.Keras is used to make the CNN.
- The link of the research paper: https://www.researchgate.net/publication/312325842


### Notes:
- Another explanation: just imagine lol. Pay attention to the name "batch_size" - the number of the inputs.
- My explanation:
   - If you convert from a **list** to a **Numpy array**, the dimension will + 1 (the newly added dimension is equal to the size of the list, usually called batch - aka number of images). 
   - If you convert from a **Numpy array** to a **Numpy array**, nothing changes lmao. Instead, use np.expand_dims(nparr, axis = 0)
- When you convert inputs (the 3D numpy array) to a NumPy array, the 3D array naturally becomes 4D:
```
inputs_array = np.array(inputs)  
print(inputs_array.shape)  
```
- Output: (num_samples, 15, 15, 3)

- Remember, everytime you apply convolutional layer, the size of the picture is decreased. So pay attention that the size is < the kernel. You can use padding.
- tf.keras.layers.Conv2D(abc, (x, x)) $\rightarrow$ the 'x' must be **odd**. Why? Go watch Convolution by Andrew Ng on Youtube.
- Better to use Numpy array. It's way more powerful than normal list.
- Use the xml.etree.ElementTree module to parse the XML file, numpy for numerical operations, and pandas for data handling.
- Better to use iterparse in this project. You may want to compare the difference between iterparse and parse.
- Hashing in this project: Each time I encounter an image, First need to check whether there is another picture that has the same hash (using set to find :) If not, push the hash into the set :).
- :) imread(image) turn the image into a high dimensonal array $\rightarrow$. So yeah, both inputs, outputs of CNN must be a numpy array $\rightarrow$ one-hot encoding the list first, then turn that one-hot encoded list into numpy array. (So: need a list $\rightarrow$ encode it using utils, or self encode it (you may use two or three dimensonal array and gan' no = 0, 1 cac kieu (for example in CNN_model la gan chay, con ben traffic.py la dung ham utils)) $\rightarrow$ turn into numpy array and then feed into the NN). 
- You cannot copy a list simply by typing list2 = list1, because: list2 will only be a reference to list1, and changes made in list1 will automatically also be made in list2. You can use the built-in List method copy() to copy a list or deepcopy().
- ordered means that they are indexed.
- The error you're encountering occurs because the move strings in your XML file are URL-encoded, and as a result, the split() method isn't separating the moves correctly. Let's break down what's happening and how to fix it. Whut how to know if the string inside the .xml file is URL - encoded lmao?

### Some functions you need to remember:
- moves = elem.text.strip().split():
  - elem.text $\rightarrow$ extracts the whole string out.
  - .strip() $\rightarrow$ deletes all the leading and trailing whitespaces.
  - .split() $\righarrow$ simply just "splits"... yk.

