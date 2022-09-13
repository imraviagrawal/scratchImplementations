


def heapify(A, n, i):
	largest = i
	l = 2*i+1
	r = 2*i+2

	if l<n and A[l]>A[largest]:
		largest = l

	if r<n and A[r]>A[largest]:
		largest = r

	if largest != i:
		A[largest], A[i] = A[i], A[largest]
		heapify(A, n, largest)

def heapSort(A):
	n = len(A)
	for i in range(int(n//2)-1, -1, -1):
		heapify(A, n, i)

	for i in range(n-1, -1, -1):
		# replace the zero (largest value with last) and
		# run heapify for array size n-i
		A[0], A[i] = A[i], A[0]
		heapify(A, i, 0)

# working code, nicely done 

A = [3, 9, 2, 1, 4, 5]

print("A before runnig heapify", A)
heapSort(A)
print("A after running heapify", A)