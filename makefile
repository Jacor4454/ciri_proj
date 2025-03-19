CPPFLAGS=
#-fsanitize=address -g
COMPILER=g++
OBJDIR=./objects/
TENSORDIR=./src/class_tensor/
TESTDIR=./tests/cpp_tests/

project: class_tensor.o main.o
	$(COMPILER) $(CPPFLAGS) -o a $(OBJDIR)main.o $(OBJDIR)class_tensor.o

test: test.o class_tensor.o
	$(COMPILER) $(CPPFLAGS) -Wall -g -pthread -o b $(OBJDIR)test.o $(OBJDIR)class_tensor.o /usr/lib/libgtest.a

class_tensor.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)class_tensor.o $(TENSORDIR)class_tensor.cpp

playground:
	$(COMPILER) $(CPPFLAGS) -Wall -g -pthread -o c playground.cpp /usr/lib/libgtest.a
	$(COMPILER) $(CPPFLAGS) -Wall -g -pthread -o d temp.cpp /usr/lib/libgtest.a

main.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)main.o main.cpp

test.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)test.o $(TESTDIR)test.cpp
