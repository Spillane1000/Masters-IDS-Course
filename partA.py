##
# Assignment - Part "A"
#
###

#from utilities import read_text_file_whole
from inverted_index import InvertedIndex
from semantic_space import SemanticSpace

def main():
  # Read in the sample data we used in class
  documents = read_data("./dataA")

  # Now we have the data we can create a crude inverted index using
  # a simple (inefficient!) linked list class.
  inv_ind = InvertedIndex()

  # Add the documents
  for d in documents:
    inv_ind.add_document(d)

  # Create our semantic space and SVD computation
  ss = SemanticSpace(inv_ind)

  # Fold in our query vector
  # TODO:  complete this method
  q = ss.create_query_vector("Human Computer Interaction")

  # This loop prints out the similarity scores between each document
  # and our query vector - results should match those of class.
  # TODO:  Complete cosine_with_doc method
  print("Showing similarities between our query and documents in semantic space")
  for i in range(9):
    print(inv_ind.docs[i] + ": " + '{0:0.3f}'.format(ss.cosine_with_doc(q, i)))

  print()

  # This code snippet iterates through every pair of terms in our semantic
  # space printing out the similarities (cosine) score of each pair.
  # TODO:  Complete cosine_with_term method
  print("Showing similarity between all terms with one another")
  for i in range(12):
    print("Similarity with " + inv_ind.terms[i][0])
    for j in range(12):
      print('{0: >10}'.format(inv_ind.terms[j][0]) + ": " + '{0:0.3f}'.format(ss.cosine_with_term(i, j)))
    print()

#####
#
# UTILITY FUNCTIONS BELOW - Can safely ignore
#
#####

##
# Given folder name it scans for all files ending in ".txt".
#
# The contents of each file is read in as a single string.
#
# @param - Folder to find .txt files
#
# @return - List containing ordered pairs (a, b)
#           a - corresponds to document name
#           b - corresponds to the text of the document as a string
###
def read_data(folder):
  import os

  documents = [ ]
  for file in os.listdir(folder):
    if file.endswith(".txt"):
      documents.append((file.rsplit('.', 1)[0], (read_text_file_whole(folder + "/" + file))))

  return documents

##
# Reads in an entire text file as a giant string
#
# @param fileName - Name of file to read in.
#
# @return The contents of the file as a string.
#
# Revision History:
# ~~~~~~~~~~~~~~~~~
# 07/10/2019 - Created (CJL).
###
def read_text_file_whole(fileName):
  try:
    file_obj = open(fileName, 'r')
    fileContents = file_obj.read()
  except OSError as ex:
    print("In IOErrorBlock")
    print("In method [read_text_file_whole] - " + str(ex))
    print("Out IOErrorBlock")
    return ""

  file_obj.close()

  return fileContents

if __name__ == "__main__":
  main()