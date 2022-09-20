echo "  ____    __          __   _____  _____   _____ "
echo " / __ \   \ \        / /  |  __ \|  __ \ / ____|"
echo "| |  | |_ _\ \  /\  / /_ _| |__) | |  | | (___  "
echo "| |  | | '_ \ \/  \/ / _' |  _  /| |  | |\___ \ "
echo "| |__| | | | \  /\  / (_| | | \ \| |__| |____) |"
echo " \____/|_| |_|\/  \/ \__,_|_|  \_\_____/|_____/ "
echo "Maxime Lejeune - UCLouvain - 20.09.2022     v1.0"
echo ""

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  bashPath=~/.bashrc
elif [[ "$OSTYPE" == "darwin"* ]]; then
  if [ -n "$ZSH_VERSION" ]; then
    bashPath=~/.zshrc
  elif [ -n "$BASH_VERSION" ]; then
    bashPath=~/.bash_profile
  else
    echo "Environment not recognised (currently supported: bash and zsh)"
  fi
else
  echo "OS not supported: please install manualy"
fi

# ------------------------------- #
echo ">> Setting up the OnWaRDS environment"
echo "">>"$bashPath"
echo "export ONWARDS_PATH=$(pwd)" >>"$bashPath"
echo "export PYTHONPATH=\${PYTHONPATH}:\$ONWARDS_PATH" >>"$bashPath"

ONWARDS_PATH=$(pwd)
PYTHONPATH=\${PYTHONPATH}:\$ONWARDS_PATH

echo ">> Done!"

# ------------------------------- #
echo ">> Compiling sources"
make -C $ONWARDS_PATH/onwards/lagSolver/libc/ lagSolver_c.so
echo ">> Done!"

# # ------------------------------- #

