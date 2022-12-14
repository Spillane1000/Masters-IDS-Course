##
# Simple linked list classes
#
# Nothing to see here folks
##
class Node:
  def __init__(self, data):
    self.item = data
    self.ref = None

class LinkedList:
  def __init__(self):
    # Currently Empty
    self.start_node = None

    # Current Size
    self.size = 0

  def insert_at_front(self, data):
    new_node = Node(data)
    new_node.ref = self.start_node
    self.start_node = new_node
    self.size += 1

  def insert_at_end(self, data):
    new_node = Node(data)

    if self.start_node is None:
      self.start_node = new_node
      self.size += 1
      return

    n = self.start_node
    while n.ref is not None:
      n = n.ref

    n.ref = new_node
    self.size += 1

  # Function to search the list
  #
  # We make this flexible with an optional parameter for the user
  # to include their own comparison function
  #
  # WARNING:  This function returns the *FIRST* match regardless of
  #           method used.
  def search(self, data, search_func = None):
    n = self.start_node
    while n is not None:
      if search_func is None:
        if n.item == data:
          return n.item
      else:
        if search_func(n.item, data) is True:
          return n.item
      n = n.ref

    return None

  # testing function
  def print_list(self):
    if self.start_node is None:
      print("List is empty.")
    else:
      n = self.start_node
      while n is not None:
        print(n.item)
        n = n.ref

