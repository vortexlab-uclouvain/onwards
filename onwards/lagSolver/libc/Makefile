CFLAGS = -O3

CC = gcc
CCLINK = gcc
OBJ = lagSolver.o flowModel.o wakeModel.o speedDeficit.o communicator.o

##################################

_OBJ = $(patsubst %,obj/%,$(OBJ))

default: display wakeModel.so

display: 
	@echo "  ____    __          __   _____  _____   _____ "
	@echo " / __ \   \ \        / /  |  __ \|  __ \ / ____|"
	@echo "| |  | |_ _\ \  /\  / /_ _| |__) | |  | | (___  "
	@echo "| |  | | '_ \ \/  \/ / _' |  _  /| |  | |\___ \ "
	@echo "| |__| | | | \  /\  / (_| | | \ \| |__| |____) |"
	@echo " \____/|_| |_|\/  \/ \__,_|_|  \_\_____/|_____/ "
	@echo "Maxime Lejeune - UCLouvain - 2022           v0.1"
	@echo ""
                                                 
                                                 
wakeModel.so: $(OBJ)
	$(CCLINK) $(CFLAGS) -shared -o lagSolver.so $(_OBJ) $(AFL_OBJ)

%.o: src/%.c 
	mkdir -p obj
	$(CC) $(CFLAGS) -c -o obj/$@ -Wall -Werror -fpic $^ 

.PHONY: clean

clean:
	rm obj/*.o *.so