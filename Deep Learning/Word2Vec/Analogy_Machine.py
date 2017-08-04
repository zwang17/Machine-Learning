from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf

with open('D:\\Google Drive\\Deep_Learning_Data\\Data\\Word2Vec\\text8.pickle','rb') as f:
    save = pickle.load(f)
    dictionary = save['dictionary']
    reverse_dictionary = save['reverse_dictionary']
    del save

model = 'DNN(2000000,512)'
session = tf.Session()
saver = tf.train.import_meta_graph('D:\\Google Drive\\Deep_Learning_Data\\Model\\Word2Vec\\{}\Saved.meta'.format(model))
saver.restore(session,'D:\\Google Drive\\Deep_Learning_Data\\Model\\Word2Vec\\{}\Saved'.format(model))


graph = tf.get_default_graph()
Analogy_Similarity = graph.get_tensor_by_name('Analogy_Similarity:0')
Analogy_Input = graph.get_tensor_by_name('Analogy_Input:0')

while True:
    print('Enter three words (lower case) one by one to create an analogy for the machine to find.')
    Input1 = input('The Analogy Machine will print several most likely candidates to complete the analogy.  Ex. (beijing) to (china) as (tokyo) to ? \n')
    Input2 = input('to: \n')
    Input3 = input('is as \n')
    Input1,Input2,Input3 = dictionary[Input1],dictionary[Input2],dictionary[Input3]
    Input = np.array([Input1,Input2,Input3])

    sim = session.run(Analogy_Similarity,feed_dict={Analogy_Input:Input})

    top_k = 10  # number of nearest neighbors
    nearest = (-sim[0, :]).argsort()[:top_k]
    log = 'to: '
    for k in range(top_k):
        close_word = reverse_dictionary[nearest[k]]
        log = '%s %s / ' % (log, close_word)
    print(log)
    c = input('Continue? (Input \'.\' to terminate)\n')
    if c == '.':
        break