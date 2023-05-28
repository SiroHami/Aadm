def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 