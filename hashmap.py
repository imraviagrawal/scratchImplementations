


class HashMap():
    def __init__(self, size):
        self.size = size
        self.map = [[] for _ in range(size)]


    def set_val(self, key, val):
        index = hash(key)%self.size
        bucket = self.map[index]

        found = False
        for index, items in enumerate(bucket):
            key_item, key_value = items

            if key_item == key:
                found = True
                break

        if found:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))



    def get_val(self, key):
        index = hash(key)%self.size
        bucket = self.map[index]

        found = False
        for index, items in enumerate(bucket):
            key_item, key_value = items

            if key_item == key:
                found = True
                break

        if found:
            return bucket[index][1] # return val
        else:
            bucket.append((key, val))



    def delete_val(self, key):
        index = hash(key)%self.size
        bucket = self.map[index]

        found = False
        for index, items in enumerate(bucket):
            key_item, key_value = items
            if key_item == key:
                found = True
                break

        if found:
            bucket.pop(index)
        else:
            "Not found"


