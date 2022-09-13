## implementation of the merge sort
# link
# time complexity O(nlogn), space O(n), divide and conquor


def mergeSort(array):
    if len(array)>1:
        l = 0
        r = len(array)

        mid = (l  + r)//2
        left = array[:mid]
        right = array[mid:]
        l = mergeSort(left)
        r = mergeSort(right)

        i, j, k = 0, 0, 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                array[k] = left[i]
                i += 1

            else:
                array[k] = right[j]
                j += 1

            k += 1


        while i < len(left):
            array[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            array[k] = right[j]
            j += 1
            k += 1


















#
# def mergeSort(array):
#     if len(array)>1:
#         mid = len(array)//2
#         l = array[:mid]
#         r = array[mid:]
#
#         left = mergeSort(l)
#         right = mergeSort(r)
#
#         i, j, k = 0, 0, 0
#
#         while i < len(l) and j < len(r):
#             if l[i] < r[j]:
#                 array[k] = l[i]
#                 i += 1
#
#             else:
#                 array[k] = r[j]
#                 j += 1
#
#             k += 1
#
#         while i < len(L):
#             array[k] = L[i]
#             i += 1
#             k += 1
#
#         while j < len(R):
#             array[k] = R[j]
#             j += 1
#             k += 1
# #
# #
# def mergeSort(array):
#     if len(array)>1:
#         mid = len(array)//2
#         L = array[:mid]
#         R = array[mid:]
#
#         left = mergeSort(L)
#         right = mergeSort(R)
#
#         i, j, k = 0, 0, 0
#
#         while i < len(L) and j < len(R):
#             if L[i] < R[j]:
#                 array[k] = L[i]
#                 i+=1
#             else:
#                 array[k] = R[j]
#                 j+=1
#
#             k += 1
#
#         while i < len(L):
#             array[k] = L[i]
#             i+= 1
#             k += 1
#
#         while j < len(R):
#             array[k] = R[j]
#             j+=1
#             k+=1


arr = [12, 11, 13, 5, 6, 7]

mergeSort(arr)
print(arr)


