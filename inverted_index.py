##
# InvertedIndex
#
# Inverted index.  Keeps track of terms and the documents they occur in
# as well as count (posting lists).
#
# Uses a simple linked list data structure internally.  Not super efficient
# but good for experimentation.
#
# Revision History:
# ~~~~~~~~~~~~~~~~~
# 07/10/2019 - Created (CJL).
###

from linked_list import LinkedList

class InvertedIndex:
  ##
  # Constructor
  #
  # No parameters required.  Simply creates an empty linked list and
  # documentID list for tracking of items posted to the index.
  #
  # The index itself is made up of the terms list described above.
  # Each terms list entry consists of an ordered pair.
  # term[i] = (a, b)
  #           a - Word indexed
  #           b - Linked list containing list data items
  #               Each data item is of the form (c, d)
  #               c - document identifier
  #               d - Term frequency of term a in document c.
  # The terms list is kept sorted on the term value
  #
  # The docs list is simply a list of the documentID's passed into
  # the InvertedIndex to be indexed.  This list is also kept sorted.
  # Useful for generating sorted output.
  #
  # Revision History:
  # ~~~~~~~~~~~~~~~~~
  # 07/10/2019 - Created (CJL).
  ###
  def __init__(self):
    # We need to keep track of the list of terms we are dealing with
    # Each word will be listed as a tuple
    # (x, y) = x is the word, y is a linked list containing lists
    #          The linked list data item is a list of the form [a, b]
    #          [a, b] = a - document identifier, b term frequency of term x
    #                   in document a.
    self.terms = [ ]

    # We need to keep track of the documents we have had added
    # Note, in practice you would use an ordinal value to represent the
    # document, not necessarily a text name.
    self.docs = [ ]

  ##
  # Adds a document to the InvertedIndex class.
  #
  # @param document - Ordered pair in the format (a, b)
  #                   a - Document identifier (to go in docs list)
  #                   b - String containing document content to be parsed
  #
  # Revision History:
  # ~~~~~~~~~~~~~~~~~
  # 08/10/2019 - Created (CJL).
  ###
  def add_document(self, document):
    doc_id = document[0]
    text   = document[1]

    # Tokenize and process text.  This is where any text pre-processing
    # will take place.
    p_text = self.process_text(text)
    self.docs.append(doc_id)

    # Sorts doc identifiers after every addition, in practice you'd
    # only do this after everything is added if at all
    self.docs.sort()

    # At this point we have our document identifier and a processed
    # list of terms, we can now start inserting this into our postings
    # list LinkedList structure
    for t in p_text:
      # For each term
      trm, x = self.search_for_term(t)
      if trm is None:
        # New term
        ll = LinkedList()
        ll.insert_at_end((doc_id, 1))
        self.terms.append((t, ll))
      else:
        ll = trm[1]
        # Check if document already exists in this linked list
        lt = ll.search(doc_id, lambda x, y:  x[0] == y)
        if lt is None:
          ll.insert_at_end([doc_id, 1])
        else:
          # Increment term frequency
          lt[1] += 1

    # Sort the terms in our term list
    # Sorts after every addition, in practice you'd only sort once
    # you have populated the inverted index.
    self.terms.sort(key = lambda tup: tup[0])

  ##
  # Given the state of the inverted index generate a term by document
  # matrix from its contents.
  #
  # Each row represents a term (sorted) and each column represents a
  # document (sorted on the document identifier).
  #
  # A value in the matrix A, say A[i][j], represents the number of times
  # term i (terms[i]) occurs in document j (docs[j]).
  #
  # @return A - Term by document matrix from internal inverted index
  #             structure.
  #
  # Revision History:
  # ~~~~~~~~~~~~~~~~~
  # 08/10/2019 - Created (CJL).
  ###
  def generate_term_by_doc_matrix(self):
    total_docs = self.get_total_docs()
    total_terms = self.get_total_terms()

    # We need to create a total_terms X total_docs matrix
    A = [[0.0 for i in range(total_docs)] for j in range(total_terms)]

    for t in self.terms:
      # Get terms postings linked list
      ll = t[1]
      for d in self.docs:
        # If our current document is in postings list add it to matrix
        item = ll.search(d, lambda x, y:  x[0] == y)
        if item is not None:
          i = self.terms.index(t)
          j = self.docs.index(d)
          A[i][j] = item[1]

    return A

  ##
  # This is the method that processes a string of text to be added to
  # the postings list.
  #
  # Typically much more would go on here.  This is the place where one
  # would perform operations such as:
  #  * stop word removal
  #  * lower casing of text
  #  * stemming or lemmatisation
  #  * any other case specific processing that needs to happen.
  #
  # @param text - String of text to be added to inverted index
  #
  # Revision History:
  # ~~~~~~~~~~~~~~~~~
  # 08/10/2019 - Created (CJL).
  ###
  def process_text(self, text):
    # We are just going to trivially split on whitespace
    t = text.split()

    # Then we will lowercase everything
    t = [item.lower() for item in t]

    # Any other text pre-processing would take place here

    return t

  ##
  # Crude linear search for the presence of a term in the inverted
  # index.  Just traverses the terms list for a match.
  #
  # @param t - Term to search for.
  #
  # @return w - The word ordered pair if found.  Remember this takes the
  #             form (a, b) where:
  #             a - Term we were searching for (match)
  #             b - Index of the term in the internal words list.
  #                 This is useful when generating matrix output as often
  #                 we will need to know the "row" this corresponds to.
  #         i - Index of w in the terms array.  Useful for using as an index into
  #             a term by document matrix.
  #
  # Revision History:
  # ~~~~~~~~~~~~~~~~~
  # 09/10/2019 - Created (CJL).
  ###
  def search_for_term(self, t):
    for w in self.terms:
      if w[0] == t:
        return w, self.terms.index(w)

    return None, None

  # Returns total number of terms in our inverted index
  def get_total_terms(self):
    return len(self.terms)

  # Returns total number of documents in our inverted index
  def get_total_docs(self):
    return len(self.docs)

  # Fun function for displaying to the screen what the current
  # term by document matrix would look like in the Inverted Index
  def print_term_by_doc_matrix(self):
    A = self.generate_term_by_doc_matrix()

    print('{0: >9}'.format(''), end='')
    for d in self.docs:
      print('{0: >6}'.format(d), end='')
    print()

    i = 0
    for t in self.terms:
      print('{0: >10}'.format(t[0]), end='')
      j = 0
      for d in self.docs:
        print('{0: .2f}'.format(float(A[i][j])) + " ", end='')
        j += 1
      print()

      i += 1

  # Prints the individual postings lists
  def print(self):
    for t in self.terms:
      llist = t[1]
      print(t[0])
      llist.print_list()


