 # to get the length of string
a = 'abcdefghiopqrstuvwxyz'
a
len(a)
s = str(2 ** 100)
s
print(s)
print(len(s))
type(s)
# use of index
S = 'Hello'
S[0] == 'H'
S[4] == 'o'
S[-1] == 'o'
len(S)

s = 'Hello'
t = s
t = s[2:4]
print(s)
print(t)  # prints 'll'

# Slices: subsequence
s = 'abcdefg'
print(s[1])
print(s[-1])
print(s[1:3])
print(s[1:-1])
print(s[:3])
print(s[2:])
print(s[:-1])
print(s[::2])
print(s[1::2])
print(s[::-1])

s = 'abcdefghijklm'
print(s[0:10:2])
for i in range(0, 10, 2):
    print(i, s[i])

# String methods: find() and rfind()
s = 'Helloe'
print(s.find('e'))
print(s.find('ll'))

s = 'abracadabra'
print(s.find('b'))
print(s.rfind('b'))
# replace()
print('a bar is a bar, essentially'.replace('bar', 'pub'))

print('a bar is a bar, essentially'.replace('bar', 'pub', 1))
# count()
print('Abracadabra'.count('a'))
print(('aaaaaaaaaa').count('aa'))
