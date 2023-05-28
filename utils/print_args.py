def print_args(args, logger=None):
  if logger is not None:
      logger.write("#### configurations ####")
  for k, v in vars(args).items():
      if logger is not None:
          logger.write('{}: {}\n'.format(k, v))
      else:
          print('{}: {}'.format(k, v))
  if logger is not None:
      logger.write("########################")