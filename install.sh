echo "  ____    __          __   _____  _____   _____ "
echo " / __ \   \ \        / /  |  __ \|  __ \ / ____|"
echo "| |  | |_ _\ \  /\  / /_ _| |__) | |  | | (___  "
echo "| |  | | '_ \ \/  \/ / _' |  _  /| |  | |\___ \ "
echo "| |__| | | | \  /\  / (_| | | \ \| |__| |____) |"
echo " \____/|_| |_|\/  \/ \__,_|_|  \_\_____/|_____/ "
echo "Maxime Lejeune - UCLouvain - 26.09.2022     v1.1"
echo ""

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  BASH_PATH=~/.bashrc
elif [[ "$OSTYPE" == "darwin"* ]]; then
  if [ -n "$ZSH_VERSION" ]; then
    BASH_PATH=~/.zshrc
  elif [ -n "$BASH_VERSION" ]; then
    BASH_PATH=~/.bash_profile
  else
    echo "Environment not recognised (currently supported: bash and zsh)"
    return -1
  fi
else
  echo "OS not supported: please install manualy"
  return -1
fi

# ------------------------------- #
echo ">> Configuring your OnWaRDS environment to ${BASH_PATH}."
echo "">>"$BASH_PATH"
echo "export ONWARDS_PATH=$(pwd)" >>"$BASH_PATH"
echo "export PYTHONPATH=\${PYTHONPATH}:\$ONWARDS_PATH" >>"$BASH_PATH"

export ONWARDS_PATH=$(pwd)
export PYTHONPATH=\${PYTHONPATH}:\$ONWARDS_PATH

echo ">> Done!"

# ------------------------------- #
echo ">> Compiling the Lagrangian Flow Model sources."
make -C $ONWARDS_PATH/onwards/lagSolver/libc/ lagSolver_c.so

if [ $? -eq 0 ]; then
  echo ">> Done!"
else
    echo ">> Compilation failed."
    return -1
fi
# # ------------------------------- #



