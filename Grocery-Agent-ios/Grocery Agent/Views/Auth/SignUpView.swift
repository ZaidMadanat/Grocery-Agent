import SwiftUI

struct SignUpView: View {
    @StateObject private var vm = SignUpViewModel()
    @Environment(\.dismiss) private var dismiss
    var onSuccess: () -> Void

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Create Account")
                        .font(.largeTitle.bold())
                    Text("Join us to start planning your meals and groceries.")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                VStack(spacing: 12) {
                    TextField("Full Name", text: $vm.name)
                        .textContentType(.name)
                        .textFieldStyle(.roundedBorder)

                    TextField("Username", text: $vm.username)
                        .textContentType(.username)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .textFieldStyle(.roundedBorder)

                    TextField("Email", text: $vm.email)
                        .textContentType(.emailAddress)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.emailAddress)
                        .textFieldStyle(.roundedBorder)

                    VStack(alignment: .leading, spacing: 4) {
                        SecureField("Password", text: $vm.password)
                            .textContentType(.newPassword)
                            .textFieldStyle(.roundedBorder)
                        Text("Minimum 8 characters")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    SecureField("Confirm Password", text: $vm.confirmPassword)
                        .textContentType(.newPassword)
                        .textFieldStyle(.roundedBorder)
                }

                if let err = vm.errorMessage {
                    Text(err)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                Button {
                    vm.signUp { onSuccess() }
                } label: {
                    HStack {
                        if vm.isLoading { ProgressView().tint(.white) }
                        Text(vm.isLoading ? "Creating Account…" : "Sign Up")
                            .bold()
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                }
                .buttonStyle(.borderedProminent)
                .disabled(vm.isLoading || !vm.isFormValid)

                // Already have an account
                Button {
                    dismiss()
                } label: {
                    HStack(spacing: 4) {
                        Text("Already have an account?")
                            .foregroundStyle(.secondary)
                        Text("Sign In")
                            .bold()
                    }
                }
                .padding(.top, 8)
            }
            .padding(24)
        }
        .navigationTitle("Sign Up")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    NavigationStack {
        SignUpView(onSuccess: {})
    }
}

