echo "  ____    __          __   _____  _____   _____ "
echo " / __ \   \ \        / /  |  __ \|  __ \ / ____|"
echo "| |  | |_ _\ \  /\  / /_ _| |__) | |  | | (___  "
echo "| |  | | '_ \ \/  \/ / _' |  _  /| |  | |\___ \ "
echo "| |__| | | | \  /\  / (_| | | \ \| |__| |____) |"
echo " \____/|_| |_|\/  \/ \__,_|_|  \_\_____/|_____/ "
echo "Maxime Lejeune - UCLouvain - 2022           v0.1"
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
source "$bashPath"
echo ">> Done!"

# # ------------------------------- #

