def solve(nums, k):
    low = 0
    high = len(nums) - 1
    quick_sort(nums, low, high)
    return nums[k - 1]


def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)


def partition(arr, low, high):
    p = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < p:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


if __name__ == '__main__':
    nums = [1, 3, 2, 4, 5, 7, 6, 8]
    print(solve(nums, 6))

