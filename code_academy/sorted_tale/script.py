import utils
import sorts

bookshelf = utils.load_books("books_small.csv")


def by_title_ascending(book_a, book_b):
    return book_a["title_lower"] > book_b["title_lower"]


def by_author_ascending(book_a, book_b):
    return book_a["author_lower"] > book_b["author_lower"]


for book in bookshelf:
    book["author_lower"] = book["author"].lower()
    book["title_lower"] = book["title"].lower()

sort_1 = sorts.bubble_sort(bookshelf, by_title_ascending)

for book in sort_1:
    print(book["title"])

bookshelf_v1 = bookshelf.copy()

sort_2 = sorts.bubble_sort(bookshelf, by_author_ascending)

for book in sort_2:
    print(book["author"])

bookshelf_v2 = bookshelf.copy()
sorts.quicksort(bookshelf_v2, 0, len(bookshelf_v2) - 1, by_author_ascending)

for book in bookshelf_v2:
    print(book["author"])

bookshelf = utils.load_books("books_large.csv")
for book in bookshelf:
    book["author_lower"] = book["author"].lower()
    book["title_lower"] = book["title"].lower()
sorts.quicksort(bookshelf, 0, len(bookshelf) - 1, by_author_ascending)
