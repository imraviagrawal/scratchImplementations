"""
Heap Sort Algorithm for sorting in increasing order:
1. Build a max heap from the input data.
2. At this point, the largest item is stored at the root of the heap. Replace it with the last item of the heap followed by reducing the size of heap by 1. Finally, heapify the root of tree.
3. Repeat above steps while size of heap is greater than 1.

linkL https://www.geeksforgeeks.org/heap-sort/

Time Complexity: Time complexity of heapify is O(Logn). Time complexity of createAndBuildHeap() is O(n) and overall time complexity of Heap Sort is O(nLogn).
"""

def heapify(array, n, i):
    largest = i
    left = 2*i + 1
    right = 2*i + 2

    if left < n and array[left]>array[largest]:
        largest = left
    if right<n and array[right]>array[largest]:
        largest = right

    if largest != i:
        array[i], array[largest] = array[largest], array[i]
        heapify(array, n, largest)

def heapSort(array):
    n = len(array)
    for i in range(n//2-1, -1, -1):
        heapify(array, n, i)

    for i in range(n-1, -1, -1):
        array[0], array[i] = array[i], array[0]
        heapify(array,i, 0)

# def heapify(array, n, i):
#     largest = i
#     l= 2*i + 1
#     r= 2*i + 2
#
#     # check if left child exist and greater than root
#     if l < n and array[largest] < array[l]:
#         largest = l
#
#     if r < n and array[largest] < array[r]:
#         largest = r
#
#     # change root if required
#     if largest != i:
#         array[largest], array[i] = array[i], array[largest]
#         # heapify root
#         heapify(array, n, largest)
#
# def heapsort(array):
#     n = len(array)
#
#     # heapify
#
#     for i in range(n//2-1, -1, -1):
#         heapify(array, n, i)
#
#     # extract element one by one
#
#     for i in range(n-1, 0, -1):
#         # print(array)
#         # first element is the largest element, move the largest element at the end and reduce the size of array by one
#         array[0], array[i] = array[i], array[0]
#         heapify(array, i, 0)


array = [ 12, 11, 13, 5, 6, 7]

heapSort(array)

print(array)