CPPFLAGS=
#-fsanitize=address -g
COMPILER=g++
OBJDIR=./objects/
TENSORDIR=./src/class_tensor/
LAYERDIR=./src/class_layer/
TESTDIR=./tests/cpp_tests/

project: class_tensor.o main.o
	$(COMPILER) $(CPPFLAGS) -o a $(OBJDIR)main.o $(OBJDIR)class_tensor.o

test: test.o class_tensor.o class_layer.o class_perceptron.o class_recersive.o activators.o
	$(COMPILER) $(CPPFLAGS) -Wall -g -pthread -o b $(OBJDIR)test.o $(OBJDIR)activators.o $(OBJDIR)class_tensor.o $(OBJDIR)class_layer.o $(OBJDIR)class_perceptron.o $(OBJDIR)class_recersive.o /usr/lib/libgtest.a

class_tensor.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)class_tensor.o $(TENSORDIR)class_tensor.cpp

class_layer.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)class_layer.o $(LAYERDIR)class_layer.cpp

class_perceptron.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)class_perceptron.o $(LAYERDIR)perceptron/class_perceptron.cpp

class_recersive.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)class_recersive.o $(LAYERDIR)recersive/class_recersive.cpp

activators.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)activators.o $(TENSORDIR)activators/activators.cpp

playground:
	$(COMPILER) $(CPPFLAGS) -Wall -g -pthread -o c playground.cpp /usr/lib/libgtest.a

main.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)main.o main.cpp

test.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)test.o $(TESTDIR)test.cpp
