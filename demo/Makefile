CC=gcc
# Disable optimizations/inlining so function layout mirrors source for demos.
CFLAGS=-O0 -g -fno-omit-frame-pointer -fno-inline
LDFLAGS=

all: demo_program stripped_demo symtab.txt

demo_program: demo_program.c
	$(CC) $(CFLAGS) demo_program.c -o demo_program

stripped_demo: demo_program
	strip -o demo_program_stripped demo_program

symtab.txt: demo_program
	nm -a -C demo_program > symtab.txt

clean:
	rm -f demo_program demo_program_stripped symtab.txt
