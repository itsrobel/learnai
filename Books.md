The Dataset is split into 3 files
- BX-Book-Ratings
- BX-Books
- BX-Users

since the object is to recomend a book based of an input and ranking of other books the users dataset _should_ not matter

__inital thoughts__

the program should take the input of the book title then operate a scan in the BX-Books dataset to find the ISBN I can then do one of 2 things


1) average rating taken from the BX-Book-Ratings, then evaluate the range for the other book rankings evaluate the nearest and rank based on most fiting

_or_ 
2) "plot" all of those points then find the nearest from the percentage total of books near that rating

basically the differencing beinging a sort of scatter or average method

I dont really have any other thoughts,
since the ranking is only using one marker that is a consistant 0-10 rating system I dont need to scale any values

__Things I probably will have to do__
first step figure out wtf a seed value is then do the following
- Regression 
- Find the most optimal K value
- find and remove any outliers

_note_ pretty sure I do not have to classify since I am not identifying anything about the book itself only find similar books in rating