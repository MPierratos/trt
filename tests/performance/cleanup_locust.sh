#!/bin/bash
# Cleanup script for Locust processes
echo "Cleaning up Locust processes..."
pkill -f 'locust.*locustfile' 2>/dev/null
sleep 1
pkill -9 -f 'locust.*locustfile' 2>/dev/null
echo "Cleanup complete"

