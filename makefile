CPPFLAGS=-fsanitize=address -g
COMPILER=g++
OBJDIR=./objects/
TENSORDIR=./src/class_tensor/
TESTDIR=./tests/cpp_tests/

project: class_tensor.o main.o
	$(COMPILER) $(CPPFLAGS) -o a $(OBJDIR)main.o $(TENSORDIR)class_tensor.h

test: test.o
	$(COMPILER) $(CPPFLAGS) -Wall -g -pthread -o b $(OBJDIR)test.o $(TENSORDIR)class_tensor.h /usr/lib/libgtest.a

main.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)main.o main.cpp
test.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)test.o $(TESTDIR)test.cpp
