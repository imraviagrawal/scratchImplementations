


# def partition(array, low, high):
#     i = low
#     pivot = array[high]
#
#     for j in range(low, high):
#         # partation everything smaller than pivot
#         if array[j] < pivot:
#             array[i], array[j] = array[j], array[i]
#             i += 1
#     array[i], array[high] = array[high], array[i]
#     return i


def partition(array, low, high):
    i = low
    pivot = array[high]

    for j in range(low, high):
        if array[j] < pivot:
            array[i], array[j] = array[j], array[i]
            i += 1

    array[i], array[high] = array[high], array[i]
    return i

def quickSort(arr, low, high):
    if low < high:
        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(arr, low, high)

    if low < high:
        pi = partition(arr, low, high)

        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)
        # Separately sort elements before
        # partition and after partition
        # quickSort(arr, low, pi - 1)
        # quickSort(arr, pi + 1, high)


arr = [12, 11, 13, 5, 6, 7]
quickSort(arr, 0, len(arr)-1)
print(arr)