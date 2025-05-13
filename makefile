CPPFLAGS=
#-fsanitize=address -g
COMPILER=g++
OBJDIR=./objects/
TENSORDIR=./src/class_tensor/
LAYERDIR=./src/class_layer/
NETWORKDIR=./src/class_network/
TESTDIR=./tests/cpp_tests/
RESTDIR=./src/rest/

project: class_tensor.o main.o
	$(COMPILER) $(CPPFLAGS) -o a $(OBJDIR)main.o $(OBJDIR)class_tensor.o

test: test.o class_tensor.o class_layers.o class_learners.o class_network.o class_learning_network.o rest.o
	$(COMPILER) $(CPPFLAGS) -Wall -g -pthread -o b $(OBJDIR)test.o $(RESTDIR)rest.o $(OBJDIR)class_learners.o $(OBJDIR)class_tensor.o $(OBJDIR)class_layers.o $(OBJDIR)class_network.o $(OBJDIR)class_learning_network.o /usr/lib/libgtest.a
	./b

clean:
	rm ./objects/*
	make clean -C $(RESTDIR)

rest.o:
	make object -C $(RESTDIR)

class_learners.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)class_learners.o $(LAYERDIR)learners/class_base.cpp

class_tensor.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)class_tensor.o $(TENSORDIR)class_tensor.cpp

class_network.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)class_network.o $(NETWORKDIR)class_network.cpp

class_learning_network.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)class_learning_network.o $(NETWORKDIR)class_learning_network.cpp

class_layers.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)class_layers.o $(LAYERDIR)class_layer.cpp

playground:
	$(COMPILER) $(CPPFLAGS) -Wall -g -pthread -o c playground.cpp /usr/lib/libgtest.a

main.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)main.o main.cpp

test.o:
	$(COMPILER) $(CPPFLAGS) -c -o $(OBJDIR)test.o $(TESTDIR)test.cpp
