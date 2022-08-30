AFL_SRC_PATH = $(AFL_PATH)/airfoilLib/src
AFL_OBJ = $(AFL_SRC_PATH)/airfoilLib.o
CFLAGS = -I $(AFL_SRC_PATH) -O3

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
	$(CCLINK) $(CFLAGS) -shared -lreadINIFile -o lagSolver.so $(_OBJ) $(AFL_OBJ)

%.o: src/%.c 
	$(CC) $(CFLAGS) -c -o obj/$@ -Wall -Werror -fpic $^ 

.PHONY: clean

clean:
	rm obj/*.o *.so