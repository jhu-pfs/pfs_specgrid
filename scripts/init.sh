#!/bin/bash

# Parse arguments

PFSSPEC_ENV="default"
INIT_GIT="1"

while (( "$#" )); do
    case "$1" in
      -e|--env)
        PFSSPEC_ENV="$2"
        shift 2
        ;;
      -g|--init-git)
        INIT_GIT="1"
        shift
        ;;
      --no-init-git)
        INIT_GIT="0"
        shift
        ;;
      --) # end argument parsing
        shift
        break
        ;;
      *) # preserve all other arguments
        PARAMS="$PARAMS $1"
        shift
        ;;
    esac
done

# Source environment configs

echo "Sourcing .bashrc"
source ~/.bashrc

echo "Sourcing environment file $PFSSPEC_ENV"
source "./configs/envs/$PFSSPEC_ENV.sh"

echo "Activating conda environment $PFSSPEC_CONDAENV"
source "$PFSSPEC_CONDAPATH/bin/activate" "$PFSSPEC_CONDAENV"

######## GIT/SSH config ########################################################

# Check if the ssh-agent is running (or vscode configured its own) and
# loads the default ssh-key to be used with github. Also needs that 
# github.com is properly configured in ~/.ssh/config

function is_vscode_ssh_agent_socket_configured() {
  if [[ "$SSH_AUTH_SOCK" == *"vscode-ssh-auth-sock"* ]]; then
    return 0
  else
    return 1
  fi
}

function is_ssh_agent_running() {
  pgrep -x ssh-agent -u $UID > /dev/null
}

function check_ssh_key() {
  ssh-add -l | grep $1 1> /dev/null 2> /dev/null
}

# Initialize SSH key for github access, if requested

if [[ $INIT_GIT -eq "1" ]]; then
  SSH_KEY=$HOME/.ssh/id_rsa
  SSH_ENV=$HOME/.ssh/environment-$HOSTNAME

  # Start ssh-agent if it's not already running
  if is_vscode_ssh_agent_socket_configured; then
    echo "vscode ssh-agent already configured"
  else
    if is_ssh_agent_running; then
      echo "ssh-agent already running"
    else
      echo "Starting ssh-agent..."
      ssh-agent -s | sed 's/^echo/#echo/' > ${SSH_ENV}
      chmod 600 ${SSH_ENV}
    fi
    source ${SSH_ENV} > /dev/null
  fi

  # Add ssh-key to agent
  if check_ssh_key $SSH_KEY; then
    echo "Github ssh-key is already added."
  else
    echo "Adding Github ssh-key, please enter passphrase."
    ssh-add $SSH_KEY
  fi
fi

################################################################################

# PFS related settings
export PYTHONPATH=.:../pysynphot:../SciScript-Python/py3

# Other settings

# PySynPhot data directories
export PYSYN_CDBS=$PFSSPEC_DATA/cdbs

# Work around issues with saving weights when running on multiple threads
export HDF5_USE_FILE_LOCKING=FALSE

# Disable tensorflow deprecation warnings
export TF_CPP_MIN_LOG_LEVEL=3

# Enable more cores for numexpr (for single process operation only!)
# export NUMEXPR_MAX_THREADS=32

# Limit number of threads (for multiprocess computation only!)
# export NUMEXPR_MAX_THREADS=12
# export OMP_NUM_THREADS=12

cd $PFSSPEC_ROOT

echo "Configured environment for PFS development."
echo "Data directory is $PFSSPEC_DATA"

pushd . > /dev/null

