output "instance_ip" {
  description = "Public IP of the eval instance"
  value       = aws_instance.irs_eval.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.irs_eval.id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i <your-key.pem> ubuntu@${aws_instance.irs_eval.public_ip}"
}
