# Compiler
CC = g++
# Compiler Flags
CFLAGS = -O3 -funroll-loops -static-libstdc++ -static-libgcc
# Output Executable Name
TARGET = benchmark
# Source Files Folder
SRC_DIR = ./Sources
# Find all .cpp files inside the source folder
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
# Convert .cpp to .o for object files
OBJS = $(SRCS:.cpp=.o)

# Default rule: Build the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Compile each .cpp file into .o
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Clean compiled files
clean:
	rm -f $(TARGET) $(OBJS)


