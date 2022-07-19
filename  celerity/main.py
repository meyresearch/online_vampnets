import click


from .vampnets import run


@click.group()
def cli():
    """
    mcelerate: fit deep Markov State Models for molecular kinetics using acclerated training methods. 
    """
    pass

@cli.command()
@click.option('-c', '--config', help="Path to a configuration file (yaml)")
@click.option('-m', '--model', help="Path to a PyTorch model")
@click.option('-t', '--trajectories', nargs=-1, help='Trajectories  - either a glob string or space separated list of trajectories')
def create(config, model, trajectories):
    
