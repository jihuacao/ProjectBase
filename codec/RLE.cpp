bool IsRepeat3(unsigned char *in, int rest)
{
    if (rest<2) return false;
    else {
        if (*in == *(in + 1) && *in == *(in + 2)) return true;
        else return false;
    }
}

int GetNoRepeat3(unsigned char *in, int rest)
{
    if (rest <= 2)
        return rest + 1;
    else {
        int c = 0,
            restc = rest;
        unsigned char *g = in;
        while (!IsRepeat3(g, restc))
        {
            g++;
            restc--;
            c++;
            if (c >= 128)
                return c;
            if (restc == 0)
                return c + 1;

        }
        return c;
    }
}



int Rle_Encode(unsigned char *inbuf, int insize, unsigned char *outbuf1, int outsize)
{
    unsigned char *src = inbuf;
    unsigned char *outbuf = outbuf1;
    int rest = insize - 1;
    int outrest = outsize;
    int count = -1;
    int flag = 0;
    while (rest >= 0)
    {
        flag = 0;
        count = -1;
        if (IsRepeat3(src, rest))
        {
            while (rest >= 0)
            {
                if (count == 127) break;
                if (*src == *(src + 1)) {
                    rest--;
                    count++;
                    src++;
                }
                else {
                    count++;
                    if (count == 127) {
                        flag = 1;
                    }
                    break;
                }
            }
            if (outrest<2)
                return -1;
            *outbuf = count | 128;
            outbuf++;
            *outbuf = *src;
            outbuf++;
            outrest -= 2;

            if (count != 127||flag==1) {
                src++;
                rest--;
            }

        }
        else
        {
            if (IsRepeat3(src, rest))
                continue;
            int num = GetNoRepeat3(src, rest);
            int i;
            if (outrest<(num + 1))
                return -1;
            *outbuf = num-1;
            outbuf++;
            for (i = 0; i<num; i++) {
                *outbuf = *(src + i);
                outbuf++;
            }
            src += num;
            rest -= num;
            outrest -= num + 1;
        }
    }
    return outsize - outrest;
}

int Rle_Decode(unsigned char *inbuf, int insize, unsigned char *outbuf, int outsize)
{
    int inrest = insize;
    int outrest = outsize;
    int i;
    unsigned char *in = inbuf;
    unsigned char *out = outbuf;
    int  ns;
    unsigned char tmp;
    while (inrest >= 0)
    {
        ns = *in+1;
        if (ns>129) {
            if ((outrest - ns + 128)<0)
                return -1;
            tmp = *(in + 1);
            for (i = 0; i<ns - 128; i++) {
                *out = tmp;
                out++;
            }
            in += 2;
            inrest -= 2;
            outrest -= ns - 128;

        }
        else {
            if ((outrest - ns)<0)
                return -1;
            in++;
            for (i = 0; i<ns; i++) {
                *out = *in;
                out++;
                in++;
            }
            inrest -= 1 + ns;
            outrest -= ns;
        }
    }
    return outsize - outrest;
}
