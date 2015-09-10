#ifndef MYRAND_H
#define MYRAND_H

// implementation of cstdlib
unsigned long int mynext = 1;
/* rand: return pseudo-random integer on 0..32767 */
int myrand(void)
{
	mynext = (214013u * mynext + 2531011u) % 2147483648u;
	return (unsigned int)(mynext / 65536u);
}
/* srand: set seed for rand() */
void mysrand(unsigned int seed)
{
	mynext = seed;
}

#endif
