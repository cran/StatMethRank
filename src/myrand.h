#ifndef MYRAND_H
#define MYRAND_H

// implementation of cstdlib
unsigned long int next = 1;
/* rand: return pseudo-random integer on 0..32767 */
int myrand(void)
{
	next = (214013u * next + 2531011u) % 2147483648u;
	return (unsigned int)(next / 65536u);
}
/* srand: set seed for rand() */
void mysrand(unsigned int seed)
{
	next = seed;
}

#endif
