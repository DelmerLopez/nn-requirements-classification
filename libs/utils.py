class Utils:

    def argmax(self, arr):
        max_val = 0
        index = 0
        for i in range(len(arr)):
            if arr[i] > max_val:
                max_val = arr[i]
                index = i
        return index