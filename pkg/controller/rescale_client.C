#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ccs-client.h"

#define SHRINK 1
#define EXPAND 0
#define OP EXPAND

#define BUF 255

int main (int argc, char **argv)
{
    int OLDNPROCS, NEWNPROCS;

    if (argc < 5) {
        printf("Usage: %s <hostname> <port> <oldprocs> <newprocs> \n", argv[0]);
        return 1;
    }

    // Create a CcsServer and connect to the given hostname and port
    CcsServer server;
    char host[BUF], *bitmap;
    int i, port, cmdLen, mode;

    sprintf(host, "%s", argv[1]);
    sscanf(argv[2], "%d", &port);
    sscanf(argv[3], "%d", &OLDNPROCS);
    sscanf(argv[4], "%d", &NEWNPROCS);

    //printf("Rescaling from %i to %i\n", OLDNPROCS, NEWNPROCS);

    if( NEWNPROCS > OLDNPROCS)
        mode = EXPAND;
    else if(OLDNPROCS > NEWNPROCS)
        mode = SHRINK;
    else{
        printf("0");
        return 0;
    }
    //printf("Connecting to server %s %d\n", host, port);
    if (CcsConnect(&server, host, port, NULL) == -1) {
        printf("0");
        return 0;
    }
    //printf("Connected to server\n");

    cmdLen = OLDNPROCS * sizeof(char) + sizeof(int) + sizeof(char);
    bitmap = (char *) malloc(cmdLen);

    if (mode == EXPAND) {
        //printf("Sending expand command.\n");
        for (i = 0; i < OLDNPROCS; i++) {
            bitmap[i] = 1;
        }
    }
    else {
        //printf("Sending shrink command.\n");
        for (i = 0; i < OLDNPROCS; i++) {
            if (i < NEWNPROCS)
                bitmap[i] = 1;
            else
                bitmap[i] = 0;
        }
    }
    memcpy(&bitmap[OLDNPROCS], &NEWNPROCS, sizeof(int));
    bitmap[OLDNPROCS+sizeof(int)] = '\0';
    if (CcsSendRequest(&server, "set_bitmap", 0, cmdLen, bitmap) == -1) {
        printf("0");
        return 0;
    }

    //printf("Waiting for reply...\n" );
    if (CcsRecvResponse(&server, cmdLen, bitmap , 180) == -1) {
        printf("0");
        return 0;
    }
    //printf("Reply received.\n");
    printf("1");

    return 0;
}
